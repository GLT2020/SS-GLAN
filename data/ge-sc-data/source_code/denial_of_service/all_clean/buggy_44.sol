pragma solidity ^0.5.0;
contract EventMetadata {
  bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
  event MetadataSet(bytes metadata);
    function _setMetadata(bytes memory metadata) internal {
        emit MetadataSet(metadata);
    }
function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
}
contract Operated {
  function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
  address private _operator;
  function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
  bool private _status;
  function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
  event OperatorUpdated(address operator, bool status);
    function _setOperator(address operator) internal {
        require(_operator != operator, "cannot set same operator");
        _operator = operator;
        emit OperatorUpdated(operator, hasActiveOperator());
    }
function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
    function _transferOperator(address operator) internal {
        require(_operator != address(0), "operator not set");
        _setOperator(operator);
    }
function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
    function _renounceOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _operator = address(0);
        _status = false;
        emit OperatorUpdated(address(0), false);
    }
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }
    function _activateOperator() internal {
        require(!hasActiveOperator(), "only when operator not active");
        _status = true;
        emit OperatorUpdated(_operator, true);
    }
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }
    function _deactivateOperator() internal {
        require(hasActiveOperator(), "only when operator active");
        _status = false;
        emit OperatorUpdated(_operator, false);
    }
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}
    function getOperator() public view returns (address operator) {
        operator = _operator;
    }
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
    function isOperator(address caller) public view returns (bool ok) {
        return (caller == getOperator());
    }
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
    function hasActiveOperator() public view returns (bool ok) {
        return _status;
    }
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
    function isActiveOperator(address caller) public view returns (bool ok) {
        return (isOperator(caller) && hasActiveOperator());
    }
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
}
contract ProofHashes {
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event HashFormatSet(uint8 hashFunction, uint8 digestSize);
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event HashSubmitted(bytes32 hash);
    function _setMultiHashFormat(uint8 hashFunction, uint8 digestSize) internal {
        emit HashFormatSet(hashFunction, digestSize);
    }
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
    function _submitHash(bytes32 hash) internal {
        emit HashSubmitted(hash);
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
}
contract MultiHashWrapper {
    struct MultiHash {
        bytes32 hash;
        uint8 hashFunction;
        uint8 digestSize;
    }
    function _combineMultiHash(MultiHash memory multihash) internal pure returns (bytes memory) {
        bytes memory out = new bytes(34);
        out[0] = byte(multihash.hashFunction);
        out[1] = byte(multihash.digestSize);
        uint8 i;
        for (i = 0; i < 32; i++) {
          out[i+2] = multihash.hash[i];
        }
        return out;
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
    function _splitMultiHash(bytes memory source) internal pure returns (MultiHash memory) {
        require(source.length == 34, "length of source must be 34");
        uint8 hashFunction = uint8(source[0]);
        uint8 digestSize = uint8(source[1]);
        bytes32 hash;
        assembly {
          hash := mload(add(source, 34))
        }
        return (MultiHash({
          hashFunction: hashFunction,
          digestSize: digestSize,
          hash: hash
        }));
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
}
 interface iFactory {
     event InstanceCreated(address indexed instance, address indexed creator, string initABI, bytes initData);
     function create(bytes calldata initData) external returns (address instance);
     function createSalty(bytes calldata initData, bytes32 salt) external returns (address instance);
     function getInitSelector() external view returns (bytes4 initSelector);
     function getInstanceRegistry() external view returns (address instanceRegistry);
     function getTemplate() external view returns (address template);
     function getSaltyInstance(bytes calldata, bytes32 salt) external view returns (address instance);
     function getNextInstance(bytes calldata) external view returns (address instance);
     function getInstanceCreator(address instance) external view returns (address creator);
     function getInstanceType() external view returns (bytes4 instanceType);
     function getInstanceCount() external view returns (uint256 count);
     function getInstance(uint256 index) external view returns (address instance);
     function getInstances() external view returns (address[] memory instances);
     function getPaginatedInstances(uint256 startIndex, uint256 endIndex) external view returns (address[] memory instances);
 }
contract Template {
  function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  address private _factory;
    modifier initializeTemplate() {
        _factory = msg.sender;
        uint32 codeSize;
        assembly { codeSize := extcodesize(address) }
        require(codeSize == 0, "must be called within contract constructor");
        _;
    }
    function getCreator() public view returns (address creator) {
        creator = iFactory(_factory).getInstanceCreator(address(this));
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function isCreator(address caller) public view returns (bool ok) {
        ok = (caller == getCreator());
    }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function getFactory() public view returns (address factory) {
        factory = _factory;
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
}
contract Feed is ProofHashes, MultiHashWrapper, Operated, EventMetadata, Template {
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event Initialized(address operator, bytes multihash, bytes metadata);
    function initialize(
        address operator,
        bytes memory multihash,
        bytes memory metadata
    ) public initializeTemplate() {
        if (operator != address(0)) {
            Operated._setOperator(operator);
            Operated._activateOperator();
        }
        if (multihash.length != 0) {
            MultiHashWrapper.MultiHash memory multihashObj = MultiHashWrapper._splitMultiHash(multihash);
            ProofHashes._setMultiHashFormat(multihashObj.hashFunction, multihashObj.digestSize);
            ProofHashes._submitHash(multihashObj.hash);
        }
        if (metadata.length != 0) {
            EventMetadata._setMetadata(metadata);
        }
        emit Initialized(operator, multihash, metadata);
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function submitHash(bytes32 multihash) public {
        require(Template.isCreator(msg.sender) || Operated.isActiveOperator(msg.sender), "only active operator or creator");
        ProofHashes._submitHash(multihash);
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function setMetadata(bytes memory metadata) public {
        require(Template.isCreator(msg.sender) || Operated.isActiveOperator(msg.sender), "only active operator or creator");
        EventMetadata._setMetadata(metadata);
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function transferOperator(address operator) public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._transferOperator(operator);
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function renounceOperator() public {
        require(Operated.isActiveOperator(msg.sender), "only active operator");
        Operated._renounceOperator();
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
}