pragma solidity >=0.5.11;
interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}
interface Marmo {
    function signer() external view returns (address _signer);
}
library ECDSA {
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        if (signature.length != 65) {
            return (address(0));
        }
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(signature, 0x20))
            s := mload(add(signature, 0x40))
            v := byte(0, mload(add(signature, 0x60)))
        }
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) {
            return address(0);
        }
        if (v != 27 && v != 28) {
            return address(0);
        }
        return ecrecover(hash, v, r, s);
    }
}
contract Ownable {
  function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
  address private _owner;
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function owner() public view returns (address) {
        return _owner;
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    modifier onlyOwner() {
        require(isOwner(), "Ownable: caller is not the owner");
        _;
    }
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
}
contract ReentrancyGuard {
  function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
  uint256 private _guardCounter;
    constructor () internal {
        _guardCounter = 1;
    }
function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
    modifier nonReentrant() {
        _guardCounter += 1;
        uint256 localCounter = _guardCounter;
        _;
        require(localCounter == _guardCounter, "ReentrancyGuard: reentrant call");
    }
}
contract FeeTransactionManager is Ownable, ReentrancyGuard {
  function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
  IERC20 public token;
  function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
  address public relayer;
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event NewRelayer(address _oldRelayer, address _newRelayer);
    constructor (address _tokenAddress, address _relayer) public {
        relayer = _relayer;
        token = IERC20(_tokenAddress);
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
    function execute(
        address _to, 
        uint256 _value, 
        uint256 _fee, 
        bytes calldata _signature
    ) nonReentrant external {
        require(tx.origin == relayer, "Invalid transaction origin");
        Marmo marmo = Marmo(msg.sender);
        bytes32 hash = keccak256(
            abi.encodePacked(
                _to,
                _value,
                _fee
            )
        );
        require(marmo.signer() == ECDSA.recover(hash, _signature), "Invalid signature");
        require(token.transferFrom(msg.sender, _to, _value));
        require(token.transferFrom(msg.sender, relayer, _fee));
    }
function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
    function setRelayer(address _newRelayer) onlyOwner external {
        require(_newRelayer != address(0));
        emit NewRelayer(relayer, _newRelayer);
        relayer = _newRelayer;
    }
function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
}