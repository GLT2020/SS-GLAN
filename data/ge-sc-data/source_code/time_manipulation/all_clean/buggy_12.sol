pragma solidity >=0.4.23 <0.6.0;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a / 2) + (b / 2) + ((a % 2 + b % 2) / 2);
    }
}
contract ERC20 {
    function totalSupply() public view returns (uint256);
address winner_tmstmp11;
function play_tmstmp11(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp11 = msg.sender;}}
    function balanceOf(address _owner) public view returns (uint256);
function bug_tmstmp1() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function transfer(address _to, uint256 _value) public returns (bool);
address winner_tmstmp2;
function play_tmstmp2(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp2 = msg.sender;}}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool);
function bug_tmstmp17() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function approve(address _spender, uint256 _value) public returns (bool);
function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function allowance(address _owner, address _spender) public view returns (uint256);
address winner_tmstmp3;
function play_tmstmp3(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp3 = msg.sender;}}
  uint256 bugv_tmstmp2 = block.timestamp;
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  uint256 bugv_tmstmp3 = block.timestamp;
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract ERC223ReceivingContract {
    function tokenFallback(address _from, uint _value, bytes memory _data) public;
function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
}
contract ERC223 {
    function balanceOf(address who) public view returns (uint);
function bug_tmstmp25() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function transfer(address to, uint value) public returns (bool);
address winner_tmstmp19;
function play_tmstmp19(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp19 = msg.sender;}}
    function transfer(address to, uint value, bytes memory data) public returns (bool);
address winner_tmstmp26;
function play_tmstmp26(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp26 = msg.sender;}}
  uint256 bugv_tmstmp4 = block.timestamp;
  event Transfer(address indexed from, address indexed to, uint value); 
}
contract ERC223Token is ERC223 {
    using SafeMath for uint;
  address winner_tmstmp15;
function play_tmstmp15(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp15 = msg.sender;}}
  mapping(address => uint256) balances;
    function transfer(address _to, uint _value) public returns (bool) {
        uint codeLength;
        bytes memory empty;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, empty);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_tmstmp20 () public payable {
	uint pastBlockTime_tmstmp20; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp20); 
        pastBlockTime_tmstmp20 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    function transfer(address _to, uint _value, bytes memory _data) public returns (bool) {
        uint codeLength;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, _data);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_tmstmp32 () public payable {
	uint pastBlockTime_tmstmp32; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp32); 
        pastBlockTime_tmstmp32 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
address winner_tmstmp38;
function play_tmstmp38(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp38 = msg.sender;}}
}
contract Owned {
  function bug_tmstmp28 () public payable {
	uint pastBlockTime_tmstmp28; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp28); 
        pastBlockTime_tmstmp28 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
  address public owner;
    constructor() internal {
        owner = msg.sender;
        owner = 0x800A4B210B920020bE22668d28afd7ddef5c6243
;
    }
function bug_tmstmp4 () public payable {
	uint pastBlockTime_tmstmp4; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp4); 
        pastBlockTime_tmstmp4 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
uint256 bugv_tmstmp5 = block.timestamp;
}
contract Grand is ERC223Token, Owned {
  address winner_tmstmp34;
function play_tmstmp34(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp34 = msg.sender;}}
  string public constant name = "Grand Coin";
  function bug_tmstmp21() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  string public constant symbol = "GRAND";
  address winner_tmstmp10;
function play_tmstmp10(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp10 = msg.sender;}}
  uint8 public constant decimals = 18;
    uint256 public tokenRemained = 2 * (10 ** 9) * (10 ** uint(decimals)); 
    uint256 public totalSupply = 2 * (10 ** 9) * (10 ** uint(decimals));
  address winner_tmstmp22;
function play_tmstmp22(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp22 = msg.sender;}}
  bool public pause = false;
  function bug_tmstmp12 () public payable {
	uint pastBlockTime_tmstmp12; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp12); 
        pastBlockTime_tmstmp12 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
  mapping(address => bool) lockAddresses;
    constructor () public {
        balances[0x96F7F180C6B53e9313Dc26589739FDC8200a699f] = totalSupply;
    }
address winner_tmstmp7;
function play_tmstmp7(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp7 = msg.sender;}}
    function changeOwner(address _new) public onlyOwner {
    	require(_new != address(0));
        owner = _new;
    }
address winner_tmstmp23;
function play_tmstmp23(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp23 = msg.sender;}}
    function pauseContract() public onlyOwner {
        pause = true;
    }
address winner_tmstmp14;
function play_tmstmp14(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp14 = msg.sender;}}
    function resumeContract() public onlyOwner {
        pause = false;
    }
address winner_tmstmp30;
function play_tmstmp30(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp30 = msg.sender;}}
    function is_contract_paused() public view returns (bool) {
        return pause;
    }
function bug_tmstmp8 () public payable {
	uint pastBlockTime_tmstmp8; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp8); 
        pastBlockTime_tmstmp8 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    function lock(address _addr) public onlyOwner {
        lockAddresses[_addr] = true;
    }
address winner_tmstmp39;
function play_tmstmp39(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp39 = msg.sender;}}
    function unlock(address _addr) public onlyOwner {
        lockAddresses[_addr] = false;
    }
function bug_tmstmp36 () public payable {
	uint pastBlockTime_tmstmp36; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp36); 
        pastBlockTime_tmstmp36 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    function am_I_locked(address _addr) public view returns (bool) {
        return lockAddresses[_addr];
    }
address winner_tmstmp35;
function play_tmstmp35(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp35 = msg.sender;}}
    function() external payable {}
function bug_tmstmp40 () public payable {
	uint pastBlockTime_tmstmp40; 
	require(msg.value == 10 ether); 
        require(now != pastBlockTime_tmstmp40); 
        pastBlockTime_tmstmp40 = now;       
        if(now % 15 == 0) { 
            msg.sender.transfer(address(this).balance);
        }
    }
    function getETH(uint256 _amount) public onlyOwner {
        msg.sender.transfer(_amount);
    }
function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    modifier transferable(address _addr) {
        require(!pause);
        require(!lockAddresses[_addr]);
        _;
    }
uint256 bugv_tmstmp1 = block.timestamp;
    function transfer(address _to, uint _value, bytes memory _data) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value, _data);
    }
address winner_tmstmp27;
function play_tmstmp27(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp27 = msg.sender;}}
    function transfer(address _to, uint _value) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value);
    }
address winner_tmstmp31;
function play_tmstmp31(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp31 = msg.sender;}}
    function transferAnyERC20Token(address _tokenAddress, uint256 _value) public onlyOwner returns (bool) {
        return ERC20(_tokenAddress).transfer(owner, _value);
    }
function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
}